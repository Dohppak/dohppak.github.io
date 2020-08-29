/* eslint-disable max-len, react/jsx-one-expression-per-line */
import React from 'react';
import { graphql } from 'gatsby';
import PropTypes from 'prop-types';
import Helmet from 'react-helmet';
import styled from '@emotion/styled';
import Layout from '../components/layout';
import siteShape from '../shapes/site';

const brunch = <a href="https://brunch.co.kr/@seung4298">Brunch</a>;
const linkedin = <a href="https://www.linkedin.com/in/seungheon-doh-61a407142/">Linked in</a>;
const maLink = <a href="mailto:seungheon.doh@gmail.com">seungheon.doh@gmail.com</a>;
const ghLink = <a href="https://github.com/dohppak">github</a>;
const design = <a href="http://seungheondoh.com/">Design portfolio</a>;
const kaggle = <a href="https://www.kaggle.com/dohppak">kaggle</a>;
const instagram = <a href="https://www.instagram.com/doh_ppak/">instagram</a>;

// const ResumeHeader = styled.header(({ theme }) => ({
//   ...theme.centerPadding,
//   display: 'flex',
//   flexDirection: 'row',
//   justifyContent: 'space-between',
//   margin: 0,
//   '> h5': {
//     margin: '2em 0 1em 0',
//   },
// }));

const H2 = styled.h2(({ theme }) => ({
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
// const H3 = styled.h3(({ theme }) => ({
//   fontFamily: 'Times',
//   fontWeight: 'bold',
//   fontStyle: 'italic',
//   fontSize: '1.25em',
//   ...theme.centerPadding,
//   marginBottom: theme.spacing,
// }));
// const H4 = styled.h4(({ theme }) => ({
//   ...theme.centerPadding,
//   fontFamily: 'Times',
//   fontWeight: 'bold',
//   fontStyle: 'italic',
//   fontSize: '1.1em',
//   color: 'rgba(0,0,0,0.4)',
//   padding: 0,
// }));
const P = styled.p(({ theme }) => ({
  fontSize: '0.9em',
  ...theme.centerPadding,
  lineHeight: 1.7,
}));

// const ResumP = styled.p(({ theme }) => ({
//   margin: 0,
//   fontSize: '0.9em',
//   ...theme.centerPadding,
// }));
// const Ul = styled.ul(({ theme }) => ({
//   fontSize: '0.9em',
//   ...theme.centerPadding,
//   marginBottom: theme.spacing,
//   marginLeft: `${theme.spacingPx * 4}px`,
// }));

const Contact = ({ data: { site: { siteMetadata: site } } }) => (
  <Layout>
    <main>
      <Helmet>
        <title>
          CONTACT
          {' '}
          &middot;
          {' '}
          {site.title}
        </title>
      </Helmet>
      <H2>CONTACT</H2>
      <P>
      제 글을 읽으시고 생각을 공유하고 싶으신 분은 언제나 편하게 메일주시길 바랍니다.<br />
      제 메일은 다음과 같으며 {maLink}, 개발과 관련된 작업은 {ghLink}에 데이터 사이언스관련 작업은 {kaggle}에 있습니다.<br />
      디자인 작업은 여기 {design},혹은 {brunch} <br />
      저의 일상은 {instagram}, 직업적인 부분은 {linkedin}에서 확인 하실 수 있습니다.
      </P>
      <P>
        If you want to read my blog and share your thoughts,
        please feel free to email me. <br />
        My E-mail is {maLink}, and You can see my development works in {ghLink} and {kaggle}<br />
        Design works in {design}, and {brunch} <br />
        If you want to see my personal life, than follow my {instagram} and {linkedin}
      </P>
    </main>
  </Layout>
);

Contact.propTypes = {
  data: PropTypes.shape({
    site: siteShape,
  }).isRequired,
};

export default Contact;

export const aboutPageQuery = graphql`
  query ContactPageSiteMetadata {
    site {
      siteMetadata {
        title
      }
    }
  }
`;
