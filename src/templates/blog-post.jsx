/* eslint-disable react/no-danger */
import React from 'react';
import { graphql } from 'gatsby';
import PropTypes from 'prop-types';
import Helmet from 'react-helmet';
import dateformat from 'dateformat';
import ReactDisqusComments from 'react-disqus-comments';
import styled from '@emotion/styled';
import site from '../shapes/site';
import Layout from '../components/layout';
import TagsList from '../components/tags-list';
import PostNav from '../components/post-nav';
import pageContextShape from '../shapes/page-context';
import postShape from '../shapes/post';
import 'katex/dist/katex.min.css';
import 'gatsby-prismjs-dracula';
import Metatags from '../components/seo';


const Main = styled.main(({ theme }) => ({
  color: theme.textColor,
}));

const Header = styled.header(({ theme }) => ({
  ...theme.centerPadding,
  display: 'flex',
  flexDirection: 'row',
  alignItems: 'center',
  justifyContent: 'space-between',
  flexWrap: 'wrap',
  [theme.smallMedia]: {
    flexDirection: 'column',
    flexWrap: 'nowrap',
  },
}));

const HeaderTitle = styled.h1(({ theme }) => ({
  width: '85%',
  marginBottom: theme.spacing,
  [theme.smallMedia]: {
    width: '100%',
    textAlign: 'center',
    marginBottom: 0,
  },
}));

const HeaderDate = styled.time(({ theme }) => ({
  fontSize: '1em',
  width: '15%',
  textAlign: 'right',
  color: 'rgba(0,0,0,0.33)',
  [theme.smallMedia]: {
    width: '100%',
    textAlign: 'center',
  },
}));

const Footer = styled.footer(({ theme }) => ({
  ...theme.centerPadding,
}));

const PostWrap = styled.section(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  marginTop: '4em',
  marginBottom: '4em',
  lineHeight: 1.7,
  // 'blockquote': {
  //   ...theme.centerPadding,
  // },
  'li p': {
    marginBlockStart: 0,
    marginBlockEnd: 0,
  },
  '> *': {
    width: '100vw',
    wordWrap: 'break-word',
    ':not(.gatsby-highlight)': {
      ...theme.centerPadding,
    },
  },
  '.gatsby-highlight': {
    width: '74vw',
    // ...theme.centerPadding,
  },
  '.gatsby-highlight pre[class*="language-text"]': {
    backgroundColor: '#F2F2F2',
  },
  'pre .language-text': {
    backgroundColor: '#F2F2F2',
    color: '#282a36',
  },
  'code': {
    fontSize: '0.9em',
  },
  ':not(pre) > code[class*="language-"]': {
    whiteSpace: 'normal',
    background: 'none',
    color: '#333',
    fontWeight: 'blod',
  },
  '> .gatsby-highlight > pre': {
    // ...theme.centerPadding,
    paddingTop: `${theme.spacingPx * 2}px`,
    paddingBottom: `${theme.spacingPx * 2}px`,
  },
  '>ul,>ol': {
    marginLeft: `${theme.spacingPx * 4}px`,
    width: `calc(100% - ${theme.spacingPx * 4}px)`,
  },
}));

const PostNavWrap = styled.div(({ theme }) => ({
  ...theme.centerPadding,
  display: 'flex',
  justifyContent: 'space-between',
  flexDirection: 'row',
  marginTop: theme.spacing,
}));

const BlogPost = ({ data, pageContext }) => {
  const { markdownRemark: post } = data;
  const { title, siteUrl } = data.site.siteMetadata;
  const { next, prev } = pageContext;

  const pathname = post.frontmatter.path;
  const taglist = post.frontmatter.tags;
  const isProduction = process.env.NODE_ENV === 'production';
  const fullUrl = `${siteUrl}${post.frontmatter.path}`;

  return (
    <Layout>
      <Metatags
        title={title}
        url={siteUrl}
        tags={taglist}
        pathname={pathname}
      />
      <Main>
        <Helmet>
          <title>
            {post.frontmatter.title}
            {' '}
            &middot;
            {' '}
            {title}
          </title>
        </Helmet>
        <article>
          <Header>
            <HeaderTitle>
              {post.frontmatter.title}
            </HeaderTitle>
            <HeaderDate dateTime={dateformat(post.frontmatter.date, 'isoDateTime')}>
              {dateformat(post.frontmatter.date, 'mmmm d, yyyy')}
            </HeaderDate>
            <TagsList tags={post.frontmatter.tags} />
          </Header>
          <PostWrap dangerouslySetInnerHTML={{ __html: post.html }} />
          <Footer>
            {isProduction && (
              <ReactDisqusComments
                shortname="Stranger"
                identifier={post.frontmatter.path}
                title={post.frontmatter.title}
                url={fullUrl}
              />
            )}
          </Footer>
        </article>
        <PostNavWrap>
          <PostNav prev post={prev} />
          <PostNav next post={next} />
        </PostNavWrap>
      </Main>
    </Layout>
  );
};

BlogPost.propTypes = {
  data: PropTypes.shape({
    site,
    markdownRemark: postShape,
  }).isRequired,
  pageContext: pageContextShape.isRequired,
};

export default BlogPost;

export const query = graphql`
  query BlogPostByPath($refPath: String!) {
    site {
      siteMetadata {
        title
        siteUrl
      }
    }
    markdownRemark(frontmatter: { path: { eq: $refPath } }) {
      html
      frontmatter {
        date(formatString: "MMMM DD, YYYY")
        path
        tags
        title
      }
    }
  }
`;
