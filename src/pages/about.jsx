/* eslint-disable max-len, react/jsx-one-expression-per-line */
import React from 'react';
import { graphql } from 'gatsby';
import PropTypes from 'prop-types';
import Helmet from 'react-helmet';
import styled from '@emotion/styled';
import Layout from '../components/layout';
import siteShape from '../shapes/site';
import excelSetupPng from '../images/excel-setup-diagram.png';

const maLink = <a href="https://www.kaist.ac.kr/html/kr/index.html">KAIST</a>;
const ghLink = <a href="https://github.com/dohppak">my GitHub</a>;
const patsLink = <a href="http://seungheondoh.com/">Design portfolio</a>;
const cdpLink = <a href="http://cursordanceparty.com">Cursor Dance Party</a>;
const esdLink = <a href={excelSetupPng}>full setup diagram</a>;

const ResumeHeader = styled.header(({ theme }) => ({
  ...theme.centerPadding,
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'space-between',
  margin: 0,
  '> h5': {
    margin: '2em 0 1em 0',
  },
}));

const H2 = styled.h2(({ theme }) => ({
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
const H3 = styled.h3(({ theme }) => ({
  fontFamily: 'Times',
  fontWeight: 'bold',
  fontStyle: 'italic',
  fontSize: '1.25em',
  ...theme.centerPadding,
  marginBottom: theme.spacing,
}));
const H4 = styled.h4(({ theme }) => ({
  ...theme.centerPadding,
  fontFamily: 'Times',
  fontWeight: 'bold',
  fontStyle: 'italic',
  fontSize: '1.1em',
  color: 'rgba(0,0,0,0.4)',
  padding: 0,
}));
const P = styled.p(({ theme }) => ({
  fontSize: '0.9em',
  ...theme.centerPadding,
  lineHeight: 1.7,
}));

const ResumP = styled.p(({ theme }) => ({
  margin: 0,
  fontSize: '0.9em',
  ...theme.centerPadding,
}));
const Ul = styled.ul(({ theme }) => ({
  fontSize: '0.9em',
  ...theme.centerPadding,
  marginBottom: theme.spacing,
  marginLeft: `${theme.spacingPx * 4}px`,
}));

const About = ({ data: { site: { siteMetadata: site } } }) => (
  <Layout>
    <main>
      <Helmet>
        <title>
          ABOUT
          {' '}
          &middot;
          {' '}
          {site.title}
        </title>
      </Helmet>
      <H2>ABOUT</H2>
      <blockquote>
        <P>
          &quot;I have been impressed with the urgency of doing.
          Knowing is not enough, we must apply. Being willing is not enough, we must do.&quot;
          <br /> -leonardo da vinci
        </P>
      </blockquote>
      <P>
        I’m a Machine Learning Engineer, with a technical background in web front-end
        development.<br /> Now, I'm MS in Culture Technology,  {maLink}. I work
        extensively in DeepLearning specalized in NLP and Sound Data.<br />
        Take a look at {ghLink} to see my personal projects.
      </P>
      <P>
        I also enjoy Design, Making, and Visualization. You can see my work in {patsLink}
        <br />
        Feel free to take a look around and contact me with any questions.
      </P>
      <H3>Résumé</H3>

      <H4>Education</H4>
      <ResumeHeader>
        <h4>Korea Advanced Institute of Science and Technology (KAIST)&middot; South Korea</h4>
        <h5>2019.02 ~ Current</h5>
      </ResumeHeader>
      <ResumP>MSc. in Graduate School of Culture Technology </ResumP>
      <ResumeHeader>
        <h4>Ulsan National Institute of Science and Technology (UNIST)&middot; South Korea</h4>
        <h5>2014.03 ~ 2019.02</h5>
      </ResumeHeader>
      <ResumP>B.S. in School of Business administration & Industrial Design </ResumP>
      <Ul>
        <li>Specialization : DataMining and UX research</li>
        <li>Academic Performance Scholarship Recipient for every semester</li>
        <li>Core course
          <br />Data Mining (데이터마이닝-기계학습), Database (데이터베이스), Customer Behavior (소비자행동론),
          <br />UX research methodology (UX연구방법론), Contextual Design (사용자 맥락 디자인- UCD 심화), Interactive Technology (인터렉티브 기술)
        </li>
      </Ul>


      <H4>Research Interest</H4>
      <Ul>
        <li>Deep Learning in Culture Technology, Music Recommend System, Context Recognition</li>
        <li>Data Visualization, Information Design, Web Design</li>
      </Ul>
      <H4>Skills</H4>
      <Ul>
        <li>DataScience: Pytorch, Tensorflow, Tensorflow.js</li>
        <li>Database: MongoDB</li>
        <li>Web Application: React.js, p5.js</li>
        <li>Visualization: Processing, Sketch, Adobe illustration, Photoshop, Premiere</li>
        <li>Programming: Python, R</li>
      </Ul>

      <H4>Recognition</H4>
      <ResumeHeader>
        <h4>NAVER & Like-Lion HACKERTON  &middot;  South Korea</h4>
        <h5>2018.12</h5>
      </ResumeHeader>
      <ResumP>1st Prize, Professor2vec : Word embedding and measure professor similarity using paper text data.</ResumP>
      <ResumeHeader>
        <h4>UNIST NAVER UnderGraduate Poster Award  &middot;  South Korea</h4>
        <h5>2017.12</h5>
      </ResumeHeader>
      <ResumP>4th Prize, Compare UX analysis and Text-mining : Measure customer similarity using interview text data</ResumP>
      <ResumeHeader>
        <h4>Spark Design Award  &middot;  USA</h4>
        <h5>2017.09</h5>
      </ResumeHeader>
      <ResumP>Concept Design Finalist, Breezi</ResumP>

      <H4>Mooc</H4>
      <ResumeHeader>
        <h4>Deep Learning  &middot;  Udacity</h4>
        <h5>2018.06 ~ 2018.11</h5>
      </ResumeHeader>
      <ResumP>Nano Degree Program</ResumP>
      <ResumeHeader>
        <h4>Artificial Intelligence and Machine Learning1,2  &middot;  KAIST, edwith</h4>
        <h5>2018.03 ~ 2019.02</h5>
      </ResumeHeader>
      <ResumP>Prof. Moon-il-Chul | Dept.of Industrial and Systems Engineering</ResumP>
      <ResumeHeader>
        <h4>Machine Learning  &middot;  KAIST, Kmooc</h4>
        <h5>2017.12</h5>
      </ResumeHeader>
      <ResumP>Prof. Alice Oh | School of computing</ResumP>
    </main>
  </Layout>
);

About.propTypes = {
  data: PropTypes.shape({
    site: siteShape,
  }).isRequired,
};

export default About;

export const aboutPageQuery = graphql`
  query AboutPageSiteMetadata {
    site {
      siteMetadata {
        title
      }
    }
  }
`;
